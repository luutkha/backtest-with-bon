"""
Streak Breakout Strategy - adapted from the original MACrossoverBacktester.

Entry Logic:
- LONG: 5 consecutive green (bullish) candles + ATR filter
- SHORT: 5 consecutive red (bearish) candles + ATR filter

Exit: Hard SL/TP with risk-reward ratio

This strategy focuses on catching reversals after extended moves.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any
from dataclasses import dataclass, field

from backtest.signals.signal_generator import (
    Signal, SignalType, SignalGenerator, SignalData,
    sma, ema, rsi, macd, bollinger_bands, atr, adx,
    cross_above, cross_below, volume_weighted_price
)


@dataclass
class StreakConfig:
    """Configuration for streak breakout strategy"""
    # ATR filter window
    atr_window_min: float = 1.0
    atr_window_max: float = 6.0

    # ATR entry offset (multiplier for entry price adjustment)
    short_atr_multiplier: float = 1.75
    long_atr_multiplier: float = 1.75

    # Risk-reward for SL/TP
    risk_reward_ratio_sl: float = 0.5  # % risk per trade (e.g., 0.5 = 0.5%)
    risk_reward_ratio_tp: float = 1.5  # TP relative to SL

    # Streak length for entry
    consecutive_candles: int = 5

    # Volume filter (optional)
    volume_ma_period: int = 20
    volume_threshold: float = 0.0  # 0 = disabled

    # ADX filter (optional)
    adx_min: float = 0.0
    adx_max: float = 100.0


def _compute_indicators(df: pd.DataFrame, config: StreakConfig) -> pd.DataFrame:
    """Pre-compute all indicators needed for streak strategy."""
    result = df.copy()

    close = result['close']
    open_ = result['open']
    high = result['high']
    low = result['low']
    volume = result.get('volume', pd.Series(0, index=close.index))

    # Candle direction and streaks
    result['is_green'] = close > open_
    result['is_red'] = close < open_

    is_green_int = result['is_green'].astype(np.int8)
    is_red_int = result['is_red'].astype(np.int8)

    # Green streak: consecutive green candles
    result['green_streak'] = is_green_int * (
        is_green_int.groupby((~result['is_green']).cumsum()).cumcount() + 1
    )
    # Red streak: consecutive red candles
    result['red_streak'] = is_red_int * (
        is_red_int.groupby((~result['is_red']).cumsum()).cumcount() + 1
    )

    # Previous streak (shifted to avoid lookahead)
    result['prev_green_streak'] = result['green_streak'].shift(1)
    result['prev_red_streak'] = result['red_streak'].shift(1)

    # ATR
    result['atr'] = atr(high, low, close, period=5)
    result['atr_pct'] = (result['atr'] / close) * 100.0

    # ATR for long-term (40 period)
    result['atr_long_term'] = atr(high, low, close, period=40)

    # Volume MA
    result['volume_ma'] = volume.rolling(window=config.volume_ma_period, min_periods=1).mean()

    # ADX
    result['adx'] = adx(high, low, close, period=10)

    # MA for trend context
    result['ma7'] = sma(close, 7)
    result['ma25'] = sma(close, 25)
    result['ma200'] = sma(close, 200) if len(close) >= 200 else pd.Series(np.nan, index=close.index)

    # Price relative to MA25
    result['ma_diff_pct'] = ((close - result['ma25']) / result['ma25']) * 100.0

    # RSI
    result['rsi'] = rsi(close, period=14)

    # Open-Close range
    result['oc_range'] = (open_ - close).abs()

    # Previous candles for context
    result['prev_close'] = close.shift(4)  # ~4 hours back
    result['prev_open'] = open_.shift(4)

    return result


def _check_long_entry(
    row: pd.Series,
    prev_row: pd.Series,
    config: StreakConfig,
    btc_row: pd.Series = None
) -> bool:
    """
    Check if LONG entry conditions are met.

    Primary: 5 consecutive green candles + ATR filter
    Optional filters (commented out in original):
    - Price below MA25
    - ADX between 30-35
    - Volume > 1.5x volume MA
    """
    # Primary: 5 consecutive green candles
    if prev_row['prev_green_streak'] != config.consecutive_candles:
        return False

    # ATR filter (primary filter in new system)
    atr_pct = row['atr_pct']
    if not (config.atr_window_min <= atr_pct <= config.atr_window_max):
        return False

    # Optional: ADX filter
    adx_val = row['adx']
    if not (np.isnan(adx_val) or config.adx_min <= adx_val <= config.adx_max):
        return False

    # Optional: Volume filter (disabled if threshold = 0)
    if config.volume_threshold > 0:
        vol_ma = row['volume_ma']
        vol = row.get('volume', 0)
        if vol_ma > 0 and vol < vol_ma * config.volume_threshold:
            return False

    # Optional: MA trend filter (price should be above MA200 for more confirmation)
    # if pd.notna(row['ma200']) and row['close'] < row['ma200']:
    #     return False

    # Optional: MA25 filter (price should be below MA25 for mean reversion)
    # if row['close'] > row['ma25']:
    #     return False

    return True


def _check_short_entry(
    row: pd.Series,
    prev_row: pd.Series,
    config: StreakConfig,
    btc_row: pd.Series = None
) -> bool:
    """
    Check if SHORT entry conditions are met.

    Primary: 5 consecutive red candles + ATR filter
    """
    # Primary: 5 consecutive red candles
    if prev_row['prev_red_streak'] != config.consecutive_candles:
        return False

    # ATR filter
    atr_pct = row['atr_pct']
    if not (config.atr_window_min <= atr_pct <= config.atr_window_max):
        return False

    # Optional: ADX filter
    adx_val = row['adx']
    if not (np.isnan(adx_val) or config.adx_min <= adx_val <= config.adx_max):
        return False

    # Optional: Volume filter
    if config.volume_threshold > 0:
        vol_ma = row['volume_ma']
        vol = row.get('volume', 0)
        if vol_ma > 0 and vol < vol_ma * config.volume_threshold:
            return False

    return True


def streak_breakout_strategy(
    h1_data: pd.DataFrame,
    **params
) -> List[Signal]:
    """
    Streak Breakout Strategy.

    Enters on reversal signals after extended consecutive moves:
    - LONG: After 5 consecutive green candles (bullish exhaustion)
    - SHORT: After 5 consecutive red candles (bearish exhaustion)

    Uses ATR-based position sizing and fixed SL/TP.

    Args:
        h1_data: DataFrame with columns [opentime, open, high, low, close, volume]
        **params: Strategy parameters (see StreakConfig)

    Returns:
        List of Signal objects
    """
    # Parse config
    config = StreakConfig(
        atr_window_min=params.get('atr_window_min', 1.0),
        atr_window_max=params.get('atr_window_max', 6.0),
        short_atr_multiplier=params.get('short_atr_multiplier', 1.75),
        long_atr_multiplier=params.get('long_atr_multiplier', 1.75),
        risk_reward_ratio_sl=params.get('risk_reward_ratio_sl', 0.5),
        risk_reward_ratio_tp=params.get('risk_reward_ratio_tp', 1.5),
        consecutive_candles=params.get('consecutive_candles', 4),
        volume_ma_period=params.get('volume_ma_period', 20),
        volume_threshold=params.get('volume_threshold', 0.0),
        adx_min=params.get('adx_min', 0.0),
        adx_max=params.get('adx_max', 100.0),
    )

    # Compute indicators
    df = _compute_indicators(h1_data.copy(), config)

    signals = []

    # Need at least 10 rows for prev_row (shift by 4 for prev_close/prev_open)
    min_idx = max(1, 4 + 1)  # Need i-1 for prev_row, and i-4 for prev_close

    for i in range(min_idx, len(df)):
        current = df.iloc[i]
        previous = df.iloc[i - 1]

        # Skip if no valid previous data
        if pd.isna(previous['prev_green_streak']) or pd.isna(previous['prev_red_streak']):
            continue

        # Skip if ATR is NaN
        if pd.isna(current['atr_pct']) or pd.isna(current['atr']):
            continue

        close = float(current['close'])
        atr = float(current['atr'])

        # Check LONG entry
        if _check_long_entry(current, previous, config):
            # Entry price adjusted by ATR
            entry_price = close - config.long_atr_multiplier * atr

            signals.append(Signal(
                timestamp=int(current['opentime']),
                signal_type=SignalType.LONG_ENTRY,
                price=float(entry_price),
                strength=1.0,
                metadata={
                    'atr_pct': float(current['atr_pct']),
                    'atr': float(atr),
                    'ma_diff_pct': float(current['ma_diff_pct']),
                    'adx': float(current['adx']) if not pd.isna(current['adx']) else 0.0,
                    'green_streak': int(current['prev_green_streak']),
                    'risk_pct': config.risk_reward_ratio_sl,
                    'rr_ratio': config.risk_reward_ratio_tp / config.risk_reward_ratio_sl,
                }
            ))

        # Check SHORT entry
        elif _check_short_entry(current, previous, config):
            entry_price = close + config.short_atr_multiplier * atr

            signals.append(Signal(
                timestamp=int(current['opentime']),
                signal_type=SignalType.SHORT_ENTRY,
                price=float(entry_price),
                strength=1.0,
                metadata={
                    'atr_pct': float(current['atr_pct']),
                    'atr': float(atr),
                    'ma_diff_pct': float(current['ma_diff_pct']),
                    'adx': float(current['adx']) if not pd.isna(current['adx']) else 0.0,
                    'red_streak': int(current['prev_red_streak']),
                    'risk_pct': config.risk_reward_ratio_sl,
                    'rr_ratio': config.risk_reward_ratio_tp / config.risk_reward_ratio_sl,
                }
            ))

    return signals


# Alias for compatibility
ma_streak_strategy = streak_breakout_strategy


# ============================================================
# Factory function for direct backtest engine integration
# ============================================================

def create_streak_breakout_backtest(
    data_dir: str,
    symbol: str,
    atr_window_min: float = 1.0,
    atr_window_max: float = 6.0,
    consecutive_candles: int = 5,
    risk_reward_ratio_sl: float = 0.5,
    risk_reward_ratio_tp: float = 1.5,
    **kwargs
):
    """
    Factory function to create a backtest with streak breakout strategy.

    Usage:
        from backtest import BacktestEngine, BacktestConfig
        from your_strategy import create_streak_breakout_backtest

        engine = create_streak_breakout_backtest(
            data_dir=r"C:\Path\\To\\Data",
            symbol="BTCUSDT",
            atr_window_min=1.0,
            atr_window_max=6.0,
            consecutive_candles=5,
        )
        results = engine.run_backtest()
    """
    from backtest import BacktestEngine, BacktestConfig
    from backtest.signals import SignalGenerator

    # Note: BacktestEngine expects a signal generator, not raw strategy function
    # This creates the appropriate configuration

    config = BacktestConfig(
        data_dir=data_dir,
        symbol=symbol,
        initial_capital=kwargs.get('initial_capital', 10000),
        tp_pct=risk_reward_ratio_sl * (risk_reward_ratio_tp / risk_reward_ratio_sl) / 100,
        sl_pct=risk_reward_ratio_sl / 100,
        leverage=kwargs.get('leverage', 1),
        verbose=kwargs.get('verbose', True),
    )

    signal_gen = SignalGenerator(strategy_func=streak_breakout_strategy)

    engine = BacktestEngine(
        config=config,
        strategy=streak_breakout_strategy,
    )

    return engine
