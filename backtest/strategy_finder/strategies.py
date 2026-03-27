"""
Strategy Templates for automated strategy finding.

Each template defines:
- Strategy function
- Parameter ranges for optimization
- Default parameters
"""

from typing import Callable, Dict, Any, List
from dataclasses import dataclass, field
import pandas as pd
import numpy as np

from ..signals.signal_generator import (
    Signal, SignalType,
    sma, ema, rsi, macd, bollinger_bands, atr, adx,
    cross_above, cross_below
)


@dataclass
class StrategyTemplate:
    """Template for a strategy with parameter ranges"""
    name: str
    description: str
    strategy_func: Callable
    param_ranges: Dict[str, List[Any]]  # Parameter name -> list of values to test
    default_params: Dict[str, Any]      # Default parameter values
    min_period: int = 1                 # Minimum warmup period needed


def build_strategy_with_params(strategy_func: Callable, params: Dict[str, Any]) -> Callable:
    """
    Wrap a strategy function with fixed parameters.

    Args:
        strategy_func: Base strategy function
        params: Parameters to fix

    Returns:
        Wrapped strategy function
    """
    def wrapped(data):
        return strategy_func(data, **params)
    return wrapped


# ============================================================
# RSI Strategy Template
# ============================================================

def rsi_strategy_template(
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

    return signals


RSI_TEMPLATE = StrategyTemplate(
    name="RSI",
    description="RSI mean reversion - Long when RSI exits oversold, Short when exits overbought",
    strategy_func=rsi_strategy_template,
    param_ranges={
        'rsi_period': [7, 10, 14, 21, 28],
        'oversold': [20, 25, 30, 35, 40],
        'overbought': [60, 65, 70, 75, 80],
    },
    default_params={
        'rsi_period': 14,
        'oversold': 30,
        'overbought': 70,
    },
    min_period=30
)


# ============================================================
# MA Crossover Strategy Template
# ============================================================

def ma_crossover_template(
    h1_data: pd.DataFrame,
    fast_period: int = 20,
    slow_period: int = 50,
    direction: str = 'both'  # 'long', 'short', 'both'
) -> List[Signal]:
    """
    Moving Average Crossover strategy.
    Long: Fast MA crosses above Slow MA
    Short: Fast MA crosses below Slow MA
    """
    df = h1_data.copy()
    df['fast_ma'] = sma(df['close'], fast_period)
    df['slow_ma'] = sma(df['close'], slow_period)

    # Calculate trend direction
    df['trend'] = (df['fast_ma'] > df['slow_ma']).astype(int)

    signals = []

    for i in range(1, len(df)):
        current = df.iloc[i]
        previous = df.iloc[i - 1]

        # Long entry: Fast MA crosses above Slow MA
        if previous['trend'] == 0 and current['trend'] == 1:
            if direction in ['long', 'both']:
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
        elif previous['trend'] == 1 and current['trend'] == 0:
            if direction in ['short', 'both']:
                signals.append(Signal(
                    timestamp=int(current['opentime']),
                    signal_type=SignalType.SHORT_ENTRY,
                    price=float(current['close']),
                    metadata={
                        'fast_ma': float(current['fast_ma']),
                        'slow_ma': float(current['slow_ma'])
                    }
                ))

    return signals


MA_CROSSOVER_TEMPLATE = StrategyTemplate(
    name="MA_CROSSOVER",
    description="MA Crossover - Long when fast MA crosses above slow MA",
    strategy_func=ma_crossover_template,
    param_ranges={
        'fast_period': [5, 10, 15, 20, 25, 30],
        'slow_period': [40, 50, 60, 70, 80, 100],
        'direction': ['both', 'long', 'short'],
    },
    default_params={
        'fast_period': 20,
        'slow_period': 50,
        'direction': 'both',
    },
    min_period=100
)


# ============================================================
# MACD Strategy Template
# ============================================================

def macd_template(
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

    # Cross detection
    df['cross'] = (df['macd'] > df['macd_signal']).astype(int)

    signals = []

    for i in range(1, len(df)):
        current = df.iloc[i]
        previous = df.iloc[i - 1]

        # Long entry: MACD crosses above signal
        if previous['cross'] == 0 and current['cross'] == 1:
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
        elif previous['cross'] == 1 and current['cross'] == 0:
            signals.append(Signal(
                timestamp=int(current['opentime']),
                signal_type=SignalType.SHORT_ENTRY,
                price=float(current['close']),
                metadata={
                    'macd': float(current['macd']),
                    'signal': float(current['macd_signal'])
                }
            ))

    return signals


MACD_TEMPLATE = StrategyTemplate(
    name="MACD",
    description="MACD - Long when MACD crosses above signal, Short when below",
    strategy_func=macd_template,
    param_ranges={
        'fast_period': [8, 12, 16, 20],
        'slow_period': [20, 26, 32, 40],
        'signal_period': [6, 9, 12, 15],
    },
    default_params={
        'fast_period': 12,
        'slow_period': 26,
        'signal_period': 9,
    },
    min_period=40
)


# ============================================================
# Bollinger Bands Strategy Template
# ============================================================

def bollinger_bands_template(
    h1_data: pd.DataFrame,
    bb_period: int = 20,
    std_dev: float = 2.0,
    mode: str = 'mean_reversion'  # 'mean_reversion' or 'breakout'
) -> List[Signal]:
    """
    Bollinger Bands strategy.
    Mean Reversion: Long when price touches lower band, Short when touches upper band
    Breakout: Long when price crosses above upper band, Short when below lower band
    """
    df = h1_data.copy()
    bb_data = bollinger_bands(df['close'], bb_period, std_dev)
    df['bb_upper'] = bb_data['upper']
    df['bb_middle'] = bb_data['middle']
    df['bb_lower'] = bb_data['lower']

    signals = []

    for i in range(1, len(df)):
        current = df.iloc[i]
        previous = df.iloc[i - 1]

        if mode == 'mean_reversion':
            # Long: Price crosses above lower band (from below)
            if previous['close'] <= previous['bb_lower'] and current['close'] > current['bb_lower']:
                signals.append(Signal(
                    timestamp=int(current['opentime']),
                    signal_type=SignalType.LONG_ENTRY,
                    price=float(current['close']),
                    metadata={'bb_lower': float(current['bb_lower'])}
                ))

            # Short: Price crosses below upper band (from above)
            elif previous['close'] >= previous['bb_upper'] and current['close'] < current['bb_upper']:
                signals.append(Signal(
                    timestamp=int(current['opentime']),
                    signal_type=SignalType.SHORT_ENTRY,
                    price=float(current['close']),
                    metadata={'bb_upper': float(current['bb_upper'])}
                ))

        elif mode == 'breakout':
            # Long: Price crosses above upper band
            if previous['close'] <= previous['bb_upper'] and current['close'] > current['bb_upper']:
                signals.append(Signal(
                    timestamp=int(current['opentime']),
                    signal_type=SignalType.LONG_ENTRY,
                    price=float(current['close']),
                    metadata={'bb_upper': float(current['bb_upper'])}
                ))

            # Short: Price crosses below lower band
            elif previous['close'] >= previous['bb_lower'] and current['close'] < current['bb_lower']:
                signals.append(Signal(
                    timestamp=int(current['opentime']),
                    signal_type=SignalType.SHORT_ENTRY,
                    price=float(current['close']),
                    metadata={'bb_lower': float(current['bb_lower'])}
                ))

    return signals


BOLLINGER_BANDS_TEMPLATE = StrategyTemplate(
    name="BOLLINGER_BANDS",
    description="Bollinger Bands - Mean reversion or breakout mode",
    strategy_func=bollinger_bands_template,
    param_ranges={
        'bb_period': [10, 15, 20, 25, 30],
        'std_dev': [1.5, 2.0, 2.5, 3.0],
        'mode': ['mean_reversion', 'breakout'],
    },
    default_params={
        'bb_period': 20,
        'std_dev': 2.0,
        'mode': 'mean_reversion',
    },
    min_period=30
)


# ============================================================
# Streak Breakout Strategy Template
# ============================================================

def streak_breakout_template(
    h1_data: pd.DataFrame,
    consecutive_candles: int = 4,
    atr_window_min: float = 1.0,
    atr_window_max: float = 6.0,
    risk_reward_ratio_sl: float = 0.5,
    risk_reward_ratio_tp: float = 1.5
) -> List[Signal]:
    """
    Streak Breakout Strategy.
    Long: After N consecutive green candles with ATR filter
    Short: After N consecutive red candles with ATR filter
    """
    df = h1_data.copy()

    close = df['close']
    open_ = df['open']
    high = df['high']
    low = df['low']

    # Candle direction
    df['is_green'] = (close > open_).astype(np.int8)
    df['is_red'] = (close < open_).astype(np.int8)

    # Green streak
    df['green_streak'] = df['is_green'] * (
        df['is_green'].groupby((~df['is_green'].astype(bool)).cumsum()).cumcount() + 1
    )
    # Red streak
    df['red_streak'] = df['is_red'] * (
        df['is_red'].groupby((~df['is_red'].astype(bool)).cumsum()).cumcount() + 1
    )

    # Previous streak (shifted to avoid lookahead)
    df['prev_green_streak'] = df['green_streak'].shift(1)
    df['prev_red_streak'] = df['red_streak'].shift(1)

    # ATR
    df['atr'] = atr(high, low, close, period=5)
    df['atr_pct'] = (df['atr'] / close) * 100.0

    signals = []

    for i in range(1, len(df)):
        current = df.iloc[i]
        previous = df.iloc[i - 1]

        # Skip if no valid previous data
        if pd.isna(previous['prev_green_streak']) or pd.isna(previous['prev_red_streak']):
            continue

        # Skip if ATR is NaN
        if pd.isna(current['atr_pct']) or pd.isna(current['atr']):
            continue

        atr_pct = current['atr_pct']

        # LONG: Consecutive green candles + ATR filter
        if (previous['prev_green_streak'] == consecutive_candles and
            atr_window_min <= atr_pct <= atr_window_max):

            signals.append(Signal(
                timestamp=int(current['opentime']),
                signal_type=SignalType.LONG_ENTRY,
                price=float(current['close']),
                metadata={
                    'atr_pct': float(atr_pct),
                    'green_streak': int(previous['prev_green_streak']),
                }
            ))

        # SHORT: Consecutive red candles + ATR filter
        elif (previous['prev_red_streak'] == consecutive_candles and
              atr_window_min <= atr_pct <= atr_window_max):

            signals.append(Signal(
                timestamp=int(current['opentime']),
                signal_type=SignalType.SHORT_ENTRY,
                price=float(current['close']),
                metadata={
                    'atr_pct': float(atr_pct),
                    'red_streak': int(previous['prev_red_streak']),
                }
            ))

    return signals


# For the streak template, we need a config-like object
class _StreakConfig:
    def __init__(self, atr_window_min, atr_window_max):
        self.atr_window_min = atr_window_min
        self.atr_window_max = atr_window_max

# Monkey-patch for streak template
def _streak_wrapper(h1_data, consecutive_candles=4, atr_window_min=1.0, atr_window_max=6.0,
                    risk_reward_ratio_sl=0.5, risk_reward_ratio_tp=1.5):
    global config
    config = _StreakConfig(atr_window_min, atr_window_max)
    return streak_breakout_template(h1_data, consecutive_candles, atr_window_min, atr_window_max,
                                     risk_reward_ratio_sl, risk_reward_ratio_tp)


STREAK_BREAKOUT_TEMPLATE = StrategyTemplate(
    name="STREAK_BREAKOUT",
    description="Streak Breakout - Long after N consecutive green, Short after N red",
    strategy_func=_streak_wrapper,
    param_ranges={
        'consecutive_candles': [3, 4, 5, 6],
        'atr_window_min': [0.5, 1.0, 1.5],
        'atr_window_max': [4.0, 6.0, 8.0],
        'risk_reward_ratio_sl': [0.3, 0.5, 0.7, 1.0],
        'risk_reward_ratio_tp': [1.0, 1.5, 2.0, 2.5],
    },
    default_params={
        'consecutive_candles': 4,
        'atr_window_min': 1.0,
        'atr_window_max': 6.0,
        'risk_reward_ratio_sl': 0.5,
        'risk_reward_ratio_tp': 1.5,
    },
    min_period=10
)


# ============================================================
# Stochastic Strategy Template
# ============================================================

def stochastic_template(
    h1_data: pd.DataFrame,
    k_period: int = 14,
    d_period: int = 3,
    oversold: float = 20,
    overbought: float = 80
) -> List[Signal]:
    """
    Stochastic Oscillator strategy.
    Long: %K crosses above %D from oversold
    Short: %K crosses below %D from overbought
    """
    df = h1_data.copy()

    from ..signals.signal_generator import stochastic
    stoch = stochastic(df['high'], df['low'], df['close'], k_period, d_period)
    df['k'] = stoch['k']
    df['d'] = stoch['d']

    signals = []

    for i in range(1, len(df)):
        current = df.iloc[i]
        previous = df.iloc[i - 1]

        # Skip NaN
        if pd.isna(current['k']) or pd.isna(current['d']):
            continue

        # Long: %K crosses above %D from oversold
        if (previous['k'] <= previous['d'] and current['k'] > current['d'] and
            current['k'] < overbought):
            signals.append(Signal(
                timestamp=int(current['opentime']),
                signal_type=SignalType.LONG_ENTRY,
                price=float(current['close']),
                metadata={'k': float(current['k']), 'd': float(current['d'])}
            ))

        # Short: %K crosses below %D from overbought
        elif (previous['k'] >= previous['d'] and current['k'] < current['d'] and
              current['k'] > oversold):
            signals.append(Signal(
                timestamp=int(current['opentime']),
                signal_type=SignalType.SHORT_ENTRY,
                price=float(current['close']),
                metadata={'k': float(current['k']), 'd': float(current['d'])}
            ))

    return signals


STOCHASTIC_TEMPLATE = StrategyTemplate(
    name="STOCHASTIC",
    description="Stochastic Oscillator - Mean reversion signals",
    strategy_func=stochastic_template,
    param_ranges={
        'k_period': [10, 14, 20],
        'd_period': [3, 5],
        'oversold': [15, 20, 25],
        'overbought': [75, 80, 85],
    },
    default_params={
        'k_period': 14,
        'd_period': 3,
        'oversold': 20,
        'overbought': 80,
    },
    min_period=20
)


# ============================================================
# Consecutive Candle Strategy (Port from Java)
# ============================================================

def consecutive_candle_strategy(
    h1_data,
    consecutive_candles: int = 5,
    atr_ratio_min: float = 0.01,
    atr_ratio_max: float = 0.06,
    atr_offset: float = 1.0,
    risk_multiplier: float = 1.0,
    use_ma50_filter: bool = True
):
    """
    Consecutive Candle Strategy - Port from Java ConsecutiveCandleCrazyStrategy.

    Entry: N consecutive green/red candles + ATR ratio filter + current candle direction
    """
    from .consecutive_candle_strategy import consecutive_candle_strategy as _ccs
    return _ccs(h1_data, consecutive_candles, atr_ratio_min, atr_ratio_max, atr_offset, risk_multiplier, use_ma50_filter)


# Import actual implementation for the template
from .consecutive_candle_strategy import consecutive_candle_strategy as _consecutive_candle_impl

CONSECUTIVE_CANDLE_TEMPLATE = StrategyTemplate(
    name="CONSECUTIVE_CANDLE",
    description="Consecutive Candle - Entry after N consecutive green/red candles with ATR filter",
    strategy_func=_consecutive_candle_impl,
    param_ranges={
        'consecutive_candles': [3, 4, 5, 6, 7],
        'atr_ratio_min': [0.005, 0.01, 0.015],
        'atr_ratio_max': [0.04, 0.06, 0.08, 0.10],
        'atr_offset': [0.5, 1.0, 1.5],
        'risk_multiplier': [0.5, 1.0, 1.5, 2.0],
    },
    default_params={
        'consecutive_candles': 5,
        'atr_ratio_min': 0.01,
        'atr_ratio_max': 0.06,
        'atr_offset': 1.0,
        'risk_multiplier': 1.0,
        'use_ma50_filter': True,
    },
    min_period=55
)


# ============================================================
# Template Registry
# ============================================================

TEMPLATES = {
    'RSI': RSI_TEMPLATE,
    'MA_CROSSOVER': MA_CROSSOVER_TEMPLATE,
    'MACD': MACD_TEMPLATE,
    'BOLLINGER_BANDS': BOLLINGER_BANDS_TEMPLATE,
    'STREAK_BREAKOUT': STREAK_BREAKOUT_TEMPLATE,
    'STOCHASTIC': STOCHASTIC_TEMPLATE,
    'CONSECUTIVE_CANDLE': CONSECUTIVE_CANDLE_TEMPLATE,
}


def get_all_templates() -> Dict[str, StrategyTemplate]:
    """Get all strategy templates"""
    return TEMPLATES.copy()


def get_template(name: str) -> StrategyTemplate:
    """Get a specific template by name"""
    return TEMPLATES.get(name)


def get_templates_by_names(names: List[str]) -> List[StrategyTemplate]:
    """Get templates by names, returns all if names is empty"""
    if not names:
        return list(TEMPLATES.values())
    return [TEMPLATES[n] for n in names if n in TEMPLATES]
