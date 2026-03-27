"""
Consecutive Candle Strategy - Port from Java ConsecutiveCandleCrazyStrategy

Entry Logic:
- LONG: N consecutive green candles + ATR ratio filter + bullish current candle
- SHORT: N consecutive red candles + ATR ratio filter + bearish current candle

Exit Logic:
- TP/SL based on ATR risk multiplier
- Trend-based (MA50) determines TP/SL direction
"""

from typing import List, Dict, Any
import pandas as pd
import numpy as np

from ..signals.signal_generator import (
    Signal, SignalType,
    sma, ema, rsi, macd, bollinger_bands, atr, adx,
    cross_above, cross_below
)


def _calculate_consecutive_candles(series: pd.Series, is_bullish: bool) -> pd.Series:
    """
    Calculate consecutive candle counts for bullish or bearish candles.

    Args:
        series: Series with boolean values (True = bullish/green)
        is_bullish: True for green candles, False for red candles

    Returns:
        Series with consecutive count (resets to 0 when candle flips)
    """
    is_target = series == is_bullish
    # Group consecutive True values and cumcount within each group
    consecutive = is_target * (
        is_target.groupby((~is_target).cumsum()).cumcount() + 1
    )
    return consecutive


def consecutive_candle_strategy(
    h1_data: pd.DataFrame,
    consecutive_candles: int = 5,
    atr_ratio_min: float = 0.01,
    atr_ratio_max: float = 0.06,
    atr_offset: float = 1.0,
    risk_multiplier: float = 1.0,
    use_ma50_filter: bool = True
) -> List[Signal]:
    """
    Consecutive Candle Strategy - Port from Java ConsecutiveCandleCrazyStrategy.

    Entry Conditions:
    - LONG:  N consecutive green candles + ATR ratio in range + current bullish candle
    - SHORT: N consecutive red candles + ATR ratio in range + current bearish candle

    Exit:
    - SL/TP based on ATR risk
    - MA50 trend filter determines TP/SL direction

    Args:
        h1_data: DataFrame with [opentime, open, high, low, close, volume]
        consecutive_candles: Number of consecutive candles for entry (default: 5)
        atr_ratio_min: Minimum ATR/close ratio (default: 0.01)
        atr_ratio_max: Maximum ATR/close ratio (default: 0.06)
        atr_offset: ATR multiplier for entry price offset (default: 1.0)
        risk_multiplier: ATR multiplier for SL/TP risk (default: 1.0)
        use_ma50_filter: Use MA50 for trend-based TP/SL (default: True)

    Returns:
        List of Signal objects
    """
    df = h1_data.copy()

    close = df['close']
    open_ = df['open']
    high = df['high']
    low = df['low']

    # Calculate ATR (5 period like Java)
    df['atr'] = atr(high, low, close, period=5)

    # Calculate ATR ratio (ATR / close)
    df['atr_ratio'] = df['atr'] / close

    # Calculate MA50
    df['ma50'] = sma(close, 50)

    # Calculate candle colors
    df['is_green'] = close > open_
    df['is_bullish'] = df['is_green']  # Current candle bullish

    # Calculate consecutive green/red streaks
    # Shift by 1 to get "previous" streak (avoid lookahead)
    df['prev_green_streak'] = _calculate_consecutive_candles(df['is_green'], True).shift(1)
    df['prev_red_streak'] = _calculate_consecutive_candles(df['is_green'], False).shift(1)

    # Also need the "current" candle streak for entry (check prev bar)
    # But Java checks consecutive from previous bars, so we need to look back

    signals = []

    # Need at least consecutive_candles + 1 bars for valid lookback
    min_idx = consecutive_candles + 2

    for i in range(min_idx, len(df)):
        current = df.iloc[i]
        prev = df.iloc[i - 1]

        # Skip if ATR is NaN
        if pd.isna(current['atr']) or pd.isna(current['atr_ratio']):
            continue

        # Skip if MA50 is NaN and filter enabled
        if use_ma50_filter and pd.isna(current['ma50']):
            continue

        atr_ratio = current['atr_ratio']
        atr_val = current['atr']
        current_close = float(current['close'])
        prev_close = float(prev['close'])

        # ATR condition check
        atr_condition = atr_ratio_min <= atr_ratio <= atr_ratio_max

        # === LONG Entry Check ===
        # Need consecutive_candles green candles before current
        # Check the streak from i-1 going backwards
        consecutive_green = 0
        for j in range(1, consecutive_candles + 1):
            if df.iloc[i - j]['is_green']:
                consecutive_green += 1
            else:
                break

        green_condition = consecutive_green == consecutive_candles
        current_bullish = current['is_bullish']

        if green_condition and atr_condition and current_bullish:
            # Calculate entry price (entry = close - atr_offset * ATR)
            entry_price = current_close - atr_offset * atr_val
            risk = risk_multiplier * atr_val

            # Determine TP/SL based on MA50
            if use_ma50_filter and current_close > current['ma50'] and prev_close > current['ma50']:
                # Price above MA50 = bullish trend = TP above entry
                tp_price = entry_price + risk  # TP above for LONG
                sl_price = entry_price - risk  # SL below entry
            else:
                # Price below MA50 = bearish/mixed = TP below entry
                tp_price = entry_price - risk  # TP below for LONG
                sl_price = entry_price + risk  # SL above entry

            signals.append(Signal(
                timestamp=int(current['opentime']),
                signal_type=SignalType.LONG_ENTRY,
                price=float(entry_price),
                metadata={
                    'consecutive_green': consecutive_green,
                    'atr_ratio': float(atr_ratio),
                    'atr': float(atr_val),
                    'ma50': float(current['ma50']) if not pd.isna(current['ma50']) else None,
                    'trend': 'above_ma50' if current_close > current['ma50'] else 'below_ma50',
                    'tp_price': float(tp_price),
                    'sl_price': float(sl_price),
                }
            ))

        # === SHORT Entry Check ===
        # Need consecutive_candles red candles before current
        consecutive_red = 0
        for j in range(1, consecutive_candles + 1):
            if not df.iloc[i - j]['is_green']:  # Red candle
                consecutive_red += 1
            else:
                break

        red_condition = consecutive_red == consecutive_candles
        current_bearish = not current['is_bullish']

        if red_condition and atr_condition and current_bearish:
            # Calculate entry price (entry = close + atr_offset * ATR)
            entry_price = current_close + atr_offset * atr_val
            risk = risk_multiplier * atr_val

            # Determine TP/SL based on MA50
            if use_ma50_filter and current_close > current['ma50'] and prev_close > current['ma50']:
                # Price above MA50 = bullish trend = TP below for SHORT
                tp_price = entry_price - risk  # TP below for SHORT
                sl_price = entry_price + risk  # SL above entry
            else:
                # Price below MA50 = bearish = TP above for SHORT
                tp_price = entry_price + risk  # TP above for SHORT
                sl_price = entry_price - risk  # SL below entry

            signals.append(Signal(
                timestamp=int(current['opentime']),
                signal_type=SignalType.SHORT_ENTRY,
                price=float(entry_price),
                metadata={
                    'consecutive_red': consecutive_red,
                    'atr_ratio': float(atr_ratio),
                    'atr': float(atr_val),
                    'ma50': float(current['ma50']) if not pd.isna(current['ma50']) else None,
                    'trend': 'above_ma50' if current_close > current['ma50'] else 'below_ma50',
                    'tp_price': float(tp_price),
                    'sl_price': float(sl_price),
                }
            ))

    return signals

