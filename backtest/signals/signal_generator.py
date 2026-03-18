"""
Signal Layer - Strategy signal generation.
Responsibilities:
- Use 1h candles only for signal generation
- Compute indicators independently
- Produce long_entry_signal, short_entry_signal, optional exit_signal
- No execution logic, no TP/SL logic, no fee logic

Output: pandas Series indexed by timestamp
"""

import pandas as pd
import numpy as np
from typing import Callable, Optional, List, Dict, Any
from dataclasses import dataclass, field
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class SignalType(Enum):
    """Types of trading signals"""
    LONG_ENTRY = "long_entry"
    SHORT_ENTRY = "short_entry"
    LONG_EXIT = "long_exit"
    SHORT_EXIT = "short_exit"
    FLAT = "flat"


@dataclass
class Signal:
    """Trading signal output"""
    timestamp: int  # milliseconds
    signal_type: SignalType
    price: float  # Reference price (close of signal candle)
    strength: float = 1.0  # Signal strength 0-1
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SignalData:
    """Container for signal series"""
    long_entry: pd.Series  # Boolean series indexed by timestamp
    short_entry: pd.Series  # Boolean series indexed by timestamp
    long_exit: Optional[pd.Series] = None  # Optional exit signals
    short_exit: Optional[pd.Series] = None


class SignalGenerator:
    """
    Base class for signal generation.
    Stateless - uses pure functions for indicator computation.
    """

    def __init__(self, strategy_func: Optional[Callable] = None):
        self.strategy_func = strategy_func

    def generate(
        self,
        h1_data: pd.DataFrame,
        params: Optional[Dict[str, Any]] = None
    ) -> SignalData:
        """
        Generate trading signals from 1h data.

        Args:
            h1_data: 1-hour OHLCV dataframe
            params: Strategy parameters

        Returns:
            SignalData with boolean series indexed by timestamp
        """
        if self.strategy_func is None:
            raise ValueError("No strategy function provided")

        params = params or {}

        # Run strategy - returns list of Signal objects
        signals = self.strategy_func(h1_data, **params)

        # Convert to SignalData format
        return self._signals_to_series(h1_data, signals)

    def _signals_to_series(
        self,
        h1_data: pd.DataFrame,
        signals: List[Signal]
    ) -> SignalData:
        """Convert list of signals to boolean series"""

        # Create empty boolean series indexed by h1 timestamps
        timestamps = h1_data['opentime'].values

        long_entry = pd.Series(False, index=pd.Index(timestamps), dtype=bool)
        short_entry = pd.Series(False, index=pd.Index(timestamps), dtype=bool)
        long_exit = pd.Series(False, index=pd.Index(timestamps), dtype=bool)
        short_exit = pd.Series(False, index=pd.Index(timestamps), dtype=bool)

        # Fill in signals
        signal_dict = {s.timestamp: s for s in signals}

        for ts in timestamps:
            if ts in signal_dict:
                sig = signal_dict[ts]
                if sig.signal_type == SignalType.LONG_ENTRY:
                    long_entry.loc[ts] = True
                elif sig.signal_type == SignalType.SHORT_ENTRY:
                    short_entry.loc[ts] = True
                elif sig.signal_type == SignalType.LONG_EXIT:
                    long_exit.loc[ts] = True
                elif sig.signal_type == SignalType.SHORT_EXIT:
                    short_exit.loc[ts] = True

        return SignalData(
            long_entry=long_entry,
            short_entry=short_entry,
            long_exit=long_exit if long_exit.any() else None,
            short_exit=short_exit if short_exit.any() else None
        )


# ===== Pure Indicator Functions =====

def sma(series: pd.Series, period: int) -> pd.Series:
    """Simple Moving Average - pure function"""
    return series.rolling(window=period, min_periods=period).mean()


def ema(series: pd.Series, period: int) -> pd.Series:
    """Exponential Moving Average - pure function"""
    return series.ewm(span=period, adjust=False, min_periods=period).mean()


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Relative Strength Index - pure function"""
    delta = series.diff()
    gain = delta.where(delta > 0, 0).rolling(window=period, min_periods=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period, min_periods=period).mean()

    rs = gain / loss
    rsi_val = 100 - (100 / (1 + rs))

    return rsi_val


def macd(
    series: pd.Series,
    fast_period: int = 12,
    slow_period: int = 26,
    signal_period: int = 9
) -> Dict[str, pd.Series]:
    """
    MACD - pure function.
    Returns dict with 'macd', 'signal', 'histogram'
    """
    ema_fast = ema(series, fast_period)
    ema_slow = ema(series, slow_period)

    macd_line = ema_fast - ema_slow
    signal_line = ema(macd_line, signal_period)
    histogram = macd_line - signal_line

    return {
        'macd': macd_line,
        'signal': signal_line,
        'histogram': histogram
    }


def bollinger_bands(
    series: pd.Series,
    period: int = 20,
    std_dev: float = 2.0
) -> Dict[str, pd.Series]:
    """
    Bollinger Bands - pure function.
    Returns dict with 'upper', 'middle', 'lower'
    """
    middle = sma(series, period)
    std = series.rolling(window=period, min_periods=period).std()

    upper = middle + (std * std_dev)
    lower = middle - (std * std_dev)

    return {
        'upper': upper,
        'middle': middle,
        'lower': lower
    }


def atr(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14
) -> pd.Series:
    """Average True Range - pure function"""
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()

    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=period, min_periods=period).mean()

    return atr


def stochastic(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    k_period: int = 14,
    d_period: int = 3
) -> Dict[str, pd.Series]:
    """Stochastic Oscillator - pure function"""
    lowest_low = low.rolling(window=k_period, min_periods=k_period).min()
    highest_high = high.rolling(window=k_period, min_periods=k_period).max()

    k = 100 * ((close - lowest_low) / (highest_high - lowest_low))
    d = k.rolling(window=d_period, min_periods=d_period).mean()

    return {'k': k, 'd': d}


def adx(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14
) -> pd.Series:
    """Average Directional Index - pure function"""
    plus_dm = high.diff()
    minus_dm = -low.diff()

    plus_dm = plus_dm.where(plus_dm > minus_dm, 0)
    minus_dm = minus_dm.where(minus_dm > plus_dm, 0)

    atr_val = atr(high, low, close, period)

    plus_di = 100 * (plus_dm.rolling(window=period, min_periods=period).mean() / atr_val)
    minus_di = 100 * (minus_dm.rolling(window=period, min_periods=period).mean() / atr_val)

    dx = 100 * ((plus_di - minus_di).abs() / (plus_di + minus_di))
    adx = dx.rolling(window=period, min_periods=period).mean()

    return adx


def volume_weighted_price(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    volume: pd.Series
) -> pd.Series:
    """Volume Weighted Price - pure function"""
    typical_price = (high + low + close) / 3
    return (typical_price * volume).rolling(window=20).sum() / volume.rolling(window=20).sum()


def cross_above(series1: pd.Series, series2: pd.Series, lookback: int = 1) -> pd.Series:
    """Detect when series1 crosses above series2"""
    above = series1 > series2
    was_below = series1.shift(lookback) <= series2.shift(lookback)
    return above & was_below


def cross_below(series1: pd.Series, series2: pd.Series, lookback: int = 1) -> pd.Series:
    """Detect when series1 crosses below series2"""
    below = series1 < series2
    was_above = series1.shift(lookback) >= series2.shift(lookback)
    return below & was_above


# ===== Example Strategy Functions =====

def rsi_strategy(
    h1_data: pd.DataFrame,
    rsi_period: int = 14,
    oversold: float = 30,
    overbought: float = 70
) -> List[Signal]:
    """
    RSI-based strategy.
    Long: RSI crosses above oversold level
    Short: RSI crosses below overbought level
    """
    df = h1_data.copy()
    df['rsi'] = rsi(df['close'], rsi_period)

    signals = []

    for i in range(1, len(df)):
        current = df.iloc[i]
        previous = df.iloc[i - 1]

        # Long entry: RSI crosses above oversold
        if previous['rsi'] <= oversold and current['rsi'] > oversold:
            signals.append(Signal(
                timestamp=int(current['opentime']),
                signal_type=SignalType.LONG_ENTRY,
                price=float(current['close']),
                metadata={'rsi': float(current['rsi'])}
            ))

        # Short entry: RSI crosses below overbought
        elif previous['rsi'] >= overbought and current['rsi'] < overbought:
            signals.append(Signal(
                timestamp=int(current['opentime']),
                signal_type=SignalType.SHORT_ENTRY,
                price=float(current['close']),
                metadata={'rsi': float(current['rsi'])}
            ))

    logger.info(f"RSI strategy generated {len(signals)} signals")
    return signals


def moving_average_crossover_strategy(
    h1_data: pd.DataFrame,
    fast_period: int = 20,
    slow_period: int = 50
) -> List[Signal]:
    """
    Moving Average Crossover strategy.
    Long: Fast MA crosses above Slow MA
    Short: Fast MA crosses below Slow MA
    """
    df = h1_data.copy()
    df['fast_ma'] = sma(df['close'], fast_period)
    df['slow_ma'] = sma(df['close'], slow_period)

    signals = []

    for i in range(1, len(df)):
        current = df.iloc[i]
        previous = df.iloc[i - 1]

        # Long entry: Fast MA crosses above Slow MA
        if previous['fast_ma'] <= previous['slow_ma'] and current['fast_ma'] > current['slow_ma']:
            signals.append(Signal(
                timestamp=int(current['opentime']),
                signal_type=SignalType.LONG_ENTRY,
                price=float(current['close']),
                metadata={
                    'fast_ma': float(current['fast_ma']),
                    'slow_ma': float(current['slow_ma'])
                }
            ))

        # Short entry: Fast MA crosses below Slow MA
        elif previous['fast_ma'] >= previous['slow_ma'] and current['fast_ma'] < current['slow_ma']:
            signals.append(Signal(
                timestamp=int(current['opentime']),
                signal_type=SignalType.SHORT_ENTRY,
                price=float(current['close']),
                metadata={
                    'fast_ma': float(current['fast_ma']),
                    'slow_ma': float(current['slow_ma'])
                }
            ))

    logger.info(f"MA Crossover strategy generated {len(signals)} signals")
    return signals


def macd_strategy(
    h1_data: pd.DataFrame,
    fast_period: int = 12,
    slow_period: int = 26,
    signal_period: int = 9
) -> List[Signal]:
    """
    MACD strategy.
    Long: MACD crosses above signal line
    Short: MACD crosses below signal line
    """
    df = h1_data.copy()
    macd_data = macd(df['close'], fast_period, slow_period, signal_period)
    df['macd'] = macd_data['macd']
    df['macd_signal'] = macd_data['signal']
    df['macd_hist'] = macd_data['histogram']

    signals = []

    for i in range(1, len(df)):
        current = df.iloc[i]
        previous = df.iloc[i - 1]

        # Long entry: MACD crosses above signal
        if previous['macd'] <= previous['macd_signal'] and current['macd'] > current['macd_signal']:
            signals.append(Signal(
                timestamp=int(current['opentime']),
                signal_type=SignalType.LONG_ENTRY,
                price=float(current['close']),
                metadata={
                    'macd': float(current['macd']),
                    'signal': float(current['macd_signal'])
                }
            ))

        # Short entry: MACD crosses below signal
        elif previous['macd'] >= previous['macd_signal'] and current['macd'] < current['macd_signal']:
            signals.append(Signal(
                timestamp=int(current['opentime']),
                signal_type=SignalType.SHORT_ENTRY,
                price=float(current['close']),
                metadata={
                    'macd': float(current['macd']),
                    'signal': float(current['macd_signal'])
                }
            ))

    logger.info(f"MACD strategy generated {len(signals)} signals")
    return signals
